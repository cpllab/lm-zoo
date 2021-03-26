Official models
==================

LM Zoo ships with a set of official models that can be queried via the
command-line tool.

If you reference any of the model names mentioned here, LM Zoo will
automatically download the relevant image from our public registry. For
example, to get token-level predictions from the ``tinylstm`` model on a file
``my_file.txt``::

  $ lm-zoo get-predictions tinylstm my_file.text out.hdf5

Note that not all models support all LM Zoo features. Check the final columns
of the table to see if the features you need are supported in each model.



Do you develop language models? please see our instructions on :ref:`contributing`.


.. raw:: html

   <script type="text/javascript">
      var registry_url = "https://cpllab.github.io/lm-zoo/registry.json";

      var all_features = {
         "tokenize": "<tt>tokenize</tt>",
         "unkify": "<tt>unkify</tt>",
         "get_surprisals": "<tt>get-surprisals</tt>",
         "get_predictions": "<tt>get-predictions</tt>",
         "mount_checkpoint": "Checkpoint mounting",
      }

      $(function(){

      // Update feature list
      var all_features_list = Object.keys(all_features);
      $("#registry-feature-header").attr("colspan", all_features_list.length);
      var registry_feature_list_htmls = $.map(all_features_list, function(name) {
         var pretty_name = all_features[name];
         return "<th id='feature-" + name + "' scope='col'>" + pretty_name + "</th>";
      })
      $("#registry-feature-list").html(registry_feature_list_htmls.join(""));

      $.getJSON(registry_url, function(data) {
         var items = $.map(data, function(registry_item, id) {
            var reference_link = "<a href='" + registry_item["ref_url"] + "'>Link</a>";

            var size_str;
            var size = registry_item["image"]["size"] / 1024;
            var next_size = size / 1024;
            var labels = ["KB", "MB", "GB"];
            var label_cur = 0;
            while (next_size > 1024) {
               size = next_size;
               next_size = size / 1024;
               label_cur += 1;
            }
            var round = function(x) { return Math.round(x * 100) / 100; }
            var size_str = label_cur == labels.length - 1 ? round(size) + " " + labels[labels.length - 1]
                                                          : round(next_size) + " " + labels[label_cur + 1];

            var columns = [id, reference_link, size_str,
                           new Date(Date.parse(registry_item["image"]["datetime"])).toLocaleDateString()];

            var feature_columns = $.map(all_features_list, function(feature) {
               var supported = registry_item["image"]["supported_features"][feature];
               var td_class = supported ? "feature-supported" : "feature-unsupported";
               var content = supported ? "Yes" : "No";
               return "<td class='" + td_class + "'>" + content + "</td>";
            });

            return "<tr><td>" + columns.join("</td><td>") + "</td>" + feature_columns.join("") + "</tr>";
         });
         console.log(items.join(""))

         $("#registry-table tbody").html(items.join(""));
         //$("#registry-table tbody").html($("#registry-table tbody").html() + items.join(""))
      })});
   </script>

   <table id="registry-table">
      <thead>
         <tr>
            <th rowspan="2" scope="col">Model name</th>
            <th rowspan="2" scope="col">Reference</th>
            <th rowspan="2" scope="col">Size</th>
            <th rowspan="2" scope="col">Last updated</th>
            <th id="registry-feature-header" colspan="5" scope="colgroup">Supported features</th>
         </tr>
         <tr id="registry-feature-list">
            <!--<th>1</th><th>2</th><th><tt>3</tt></th><th>4</th><th>5</th>-->
         </tr>
      </thead>
      <tbody></tbody>
   </table>
