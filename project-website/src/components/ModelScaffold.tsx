type Props = { title: string; module: string; plan: string };

const ModelScaffold = ({ title, module, plan }: Props) => {
  return (
    <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-24 px-6 text-[#1a1a1a]">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          <h1 className="text-4xl md:text-5xl font-medium tracking-tight">{title}</h1>
          <span className="text-xs font-bold uppercase tracking-wide bg-amber-500 text-white px-3 py-1 rounded-full">
            Coming in {module}
          </span>
        </div>
        <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">{plan}</p>
      </div>
    </div>
  );
};

export default ModelScaffold;
